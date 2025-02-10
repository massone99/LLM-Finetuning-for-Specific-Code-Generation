// Message types defined
case object EchoMessage extends Message

// Actor implementation
class EchoActor extends Actor {
  override def receive: Receive = {
    case msg @ _ => sender().tell(msg)
  }
}

// System integration
import akka.actor.ActorSystem
object EchoExample {
  def main(args: Array[String]) {
    implicit val system = ActorSystem("echo-system")

    val echoActor = systemactorOf[EchoActor]()

    // Setup test message
    val msg = EchoMessage

    // Send test message and expect response
    val responseFuture = Future {
      val receivedMsg = receiveMsg(msg)
      println(s"Received: $receivedMsg")
      receivedMsg
    }
    respondWith(responseFuture)

  }

  def respondWith[F](future: Future[F])(implicit system: ActorSystem): Unit = {
    val futureRef = system.actorRef("response-future", classOf[ResponseFuture])
    future.map { result =>
      system.actorSelection(futureRef.path).tell(result)
    }
  }
}

// Response future actor
import akka.actor.Typed
import akka.actor.typed.{Behavior, PostStop}
import akka.actor.typed.scaladsl.Behaviors

class ResponseFuture extends Actor[Typed.EventMessage[String]] {
  override def onEvent(msg: String): Behavior = Behaviors.same

  override def onPostStop(): Unit = {
    println(s"Response future actor stopped")
  }
}
