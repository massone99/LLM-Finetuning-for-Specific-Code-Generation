import akka.actor.typed.{ActorSystem, Behavior}
import akka.actor.typed.scaladsl.Behaviors
import akka.cluster.typed.{ClusterSingleton, SingletonActor}
import com.typesafe.config.ConfigFactory

object ConfigManager {
  sealed trait Command
  final case class UpdateConfig(config: String) extends Command
  case object GetConfig extends Command

  def apply(): Behavior[Command] = Behaviors.receive { (context, message) =>
    message match {
      case UpdateConfig(config) =>
        context.log.info(s"Config updated to: $config")
        Behaviors.same
      case GetConfig =>
        context.log.info("Returning current config")
        Behaviors.same
    }
  }
}

object ClusterSingletonApp extends App {
  val config = ConfigFactory.load()
  val system: ActorSystem[Nothing] = ActorSystem(Behaviors.empty, "ClusterSystem", config)

  val singletonManager = ClusterSingleton(system)
  val configManagerProxy = singletonManager.init(
    SingletonActor(
      Behaviors.supervise(ConfigManager()).onFailure[Exception](
        akka.actor.typed.SupervisorStrategy.restart
      ),
      "ConfigManager"
    )
  )

  // Interact with the singleton
  configManagerProxy ! ConfigManager.UpdateConfig("NewConfigValue")

  Thread.sleep(5000)
  system.terminate()
}