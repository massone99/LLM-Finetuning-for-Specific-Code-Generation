package sample.cluster.simple


import akka.actor.typed.scaladsl.Behaviors
import akka.actor.typed.ActorSystem
import akka.actor.typed.Behavior
import com.typesafe.config.ConfigFactory


object App {


  object RootBehavior {
    def apply(): Behavior[Nothing] = Behaviors.setup[Nothing] { context =>
      // Create an actor that handles cluster domain events
      context.spawn(ClusterListener(), "ClusterListener")


      Behaviors.empty
    }
  }


  def main(args: Array[String]): Unit = {
    val ports =
      if (args.isEmpty)
        Seq(25251, 25252, 0)
      else
        args.toSeq.map(_.toInt)
    ports.foreach(startup)
  }


  def startup(port: Int): Unit = {
    // Override the configuration of the port
    val config = ConfigFactory.parseString(
      s"""
      akka.remote.artery.canonical.port=$port
      """
    ).withFallback(ConfigFactory.load())


    // Create an Akka system
    ActorSystem[Nothing](RootBehavior(), "ClusterSystem", config)
  }


  object ClusterListener {
    import akka.actor.typed.ActorRef
    import akka.cluster.ClusterEvent
    import akka.cluster.ClusterEvent.MemberEvent
    import akka.cluster.ClusterEvent.ReachabilityEvent
    import akka.actor.typed.scaladsl.LoggerOps


    sealed trait Event
    final case class WrappedMemberEvent(event: MemberEvent) extends Event
    final case class WrappedReachabilityEvent(event: ReachabilityEvent) extends Event


    def apply(): Behavior[Event] = {
      Behaviors.setup { context =>
        val memberEventAdapter: ActorRef[MemberEvent] =
          context.messageAdapter(WrappedMemberEvent)
        val reachabilityEventAdapter: ActorRef[ReachabilityEvent] =
          context.messageAdapter(WrappedReachabilityEvent)


        val cluster = akka.cluster.typed.Cluster(context.system)
        cluster.subscriptions ! akka.cluster.typed.Subscribe(memberEventAdapter, classOf[ClusterEvent.MemberEvent])
        cluster.subscriptions ! akka.cluster.typed.Subscribe(reachabilityEventAdapter, classOf[ClusterEvent.ReachabilityEvent])


        Behaviors.receiveMessage {
          case WrappedMemberEvent(memberEvent) =>
            context.log.info2("MemberEvent: {}", memberEvent)
            Behaviors.same
          case WrappedReachabilityEvent(reachabilityEvent) =>
            context.log.info2("ReachabilityEvent: {}", reachabilityEvent)
            Behaviors.same
        }
      }
    }
  }
}